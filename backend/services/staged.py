"""Staged early-readout probes: read a probe's label half-way down its own full-fidelity
trajectory, and finish only the probes whose label may have changed.

A chord probe today renders a complete 4-step-schedule image (probe_steps = 4) -- a DIFFERENT
trajectory from the 8-step full-fidelity image -- and a certified run then re-renders the bracket
ends at full fidelity. A staged probe runs the tool's full-fidelity schedule itself (S = 8 steps,
CFG 4.0, 512 px) only up to step t, decodes the PREDICTED clean image

    x̂0,t = x_t − σ_t · v̂_t                      (v̂_t = the guided velocity of the t-th step)

reads it with the tool's DINOv2, and keeps the latent the t-th step produced. Consecutive probes
whose x̂0 readouts are at least θ apart (cosine distance) are a FLAGGED segment; both of its probes
are RESUMED from the kept latent to step S, which is exactly the full-fidelity image and label
(same trajectory, same code path -- tests/staged_parity_cpu.py). Crossings are then detected with
the Cascade's own rule (1 − cos > COS_T) on those exact labels. An unflagged segment is never
finished.

Step numbering: "step t" is the t-th transformer (CFG) evaluation, t = 1..S. x̂0 at step t is the
prediction that evaluation makes from the latent it starts at; the CACHED latent at t is the one
the t-th scheduler step produced, i.e. the input of step t + 1. Resuming from it runs steps
t + 1..S. At t = S the "readout" is the final image itself (decoded exactly as the pipeline does).

Two halves:
  * WORKER side (torch, runs inside a gpu_pool worker on its resident pipe + DINOv2):
    `StagedRecipe` prepares one point exactly as Flux2KleinPipeline.__call__ does, `denoise`
    replicates its loop between any two steps with optional per-step hooks, `decode_pil`
    replicates its final decode (unpack -> BN-denorm -> unpatchify -> VAE -> postprocess "pil"),
    `x0hat_embedding` decodes a predicted x̂0 through that same decode and the tool's DINOv2
    transform. `worker_*` functions are the bodies of the four task types.
  * MAIN-process side (numpy): the LRU of cached latents (`LatentCache`), the flagging and
    resume-selection rules, cost accounting (`Ledger`), and `readout` / `resume` dispatchers that
    follow cascade.evaluate's shard / inbox discipline. Tests stub `readout` and `resume`
    (module attributes, called as `st.readout(...)`), as tests/metro_test.py stubs evaluate.
"""
from __future__ import annotations

import io
import threading
import time
from collections import OrderedDict

import numpy as np

# Defaults. THETA is provisional: it is replaced by the h27 calibration
# (search_problem/outputs/h27_staged_probes/PREREG.md section 4).
T_DEFAULT = 4
THETA_DEFAULT = 0.10
CACHE_CAP_BYTES = 2 << 30          # 2 GiB of cached latents (~8k probes at 512 px)
CHUNK = 8                          # points per GPU per round (cascade.evaluate's CHUNK)
STALL_TIMEOUT = 15 * 60

_SEQ = [0]
_SEQ_LOCK = threading.Lock()


def _next_seq():
    with _SEQ_LOCK:
        _SEQ[0] += 1
        return _SEQ[0]


# =============================================================================================
# WORKER side
# =============================================================================================

_DTYPES = None


def _dtype_table():
    global _DTYPES
    if _DTYPES is None:
        import torch
        _DTYPES = {"bfloat16": (torch.bfloat16, torch.int16), "float16": (torch.float16, torch.int16),
                   "float32": (torch.float32, torch.int32)}
    return _DTYPES


def latent_to_bytes(lat):
    """(bytes, shape, dtype name) of a latent tensor, bit-exact (bf16 travels as its raw int16 view)."""
    name = str(lat.dtype).replace("torch.", "")
    _dt, view = _dtype_table()[name]
    arr = lat.detach().to("cpu").contiguous().view(view).numpy()
    return arr.tobytes(), tuple(int(s) for s in lat.shape), name


def bytes_to_latent(b, shape, dtype_name, device):
    """Inverse of latent_to_bytes."""
    import torch
    dt, view = _dtype_table()[dtype_name]
    t = torch.frombuffer(bytearray(b), dtype=view).view(dt).reshape(tuple(shape))
    return t.to(device)


def _sync(device):
    import torch
    if getattr(device, "type", str(device)).startswith("cuda"):
        torch.cuda.synchronize(device)


class Stopwatch:
    """Wall-clock of the denoising loop alone: hooks pause it around their own work (decode,
    DINOv2), so `elapsed()` is the cumulative denoising time through the last step."""

    def __init__(self, device):
        self.device = device
        _sync(device)
        self._t0 = time.perf_counter()
        self._acc = 0.0
        self._running = True

    def pause(self):
        if self._running:
            _sync(self.device)
            self._acc += time.perf_counter() - self._t0
            self._running = False
        return self._acc

    def resume(self):
        if not self._running:
            _sync(self.device)
            self._t0 = time.perf_counter()
            self._running = True

    def elapsed(self):
        if self._running:
            _sync(self.device)
            return self._acc + time.perf_counter() - self._t0
        return self._acc


def mix_discover(base, weights):
    """The Cascade probe's mixing, verbatim from gpu_pool._process_discover (affine, zeros skipped)."""
    return sum(float(w) * base[n] for n, w in enumerate(weights) if w != 0.0)


class StagedRecipe:
    """Everything Flux2KleinPipeline.__call__ prepares for ONE point before its denoising loop,
    built with the pipeline's own methods in the pipeline's order:

      * prompt_embeds / text_ids    -- encode_prompt(prompt_embeds=emb): repeat/view + text ids
      * negative embeds / ids       -- the worker's cached empty-prompt embedding (the Cascade's
                                       Discover path passes it; __call__ then skips the encode)
      * latents / latent_ids        -- prepare_latents with torch.Generator(device).manual_seed
      * the sigma schedule          -- linspace(1, 1/S, S), resolution shift mu =
                                       compute_empirical_mu(image_seq_len, S), retrieve_timesteps
    """

    def __init__(self, pipe, emb, seed, height, width, steps, guidance, device, neg=None):
        import torch
        self.pipe = pipe
        self.device = device
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.height, self.width = int(height), int(width)
        with torch.no_grad():
            self.emb, self.text_ids = pipe.encode_prompt(prompt=None, prompt_embeds=emb, device=device)
            self.cfg = self.guidance > 1 and not pipe.config.is_distilled
            self.neg = self.neg_ids = None
            if self.cfg:
                if neg is None:
                    neg = getattr(pipe, "_cached_neg_embeds", None)
                self.neg, self.neg_ids = pipe.encode_prompt(prompt="", prompt_embeds=neg, device=device)
            gen = torch.Generator(device=device).manual_seed(int(seed))
            in_ch = pipe.transformer.config.in_channels
            self.latents0, self.latent_ids = pipe.prepare_latents(
                batch_size=1, num_latents_channels=in_ch // 4, height=self.height, width=self.width,
                dtype=self.emb.dtype, device=device, generator=gen, latents=None)
        self.image_seq_len = self.latents0.shape[1]

    def schedule(self):
        """(timesteps, scheduler) after the pipeline's step 6. Re-run for every denoise call: the
        worker's other task types reset the shared scheduler."""
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
        sch = self.pipe.scheduler
        S = self.steps
        sigmas = np.linspace(1.0, 1 / S, S)
        if hasattr(sch.config, "use_flow_sigmas") and sch.config.use_flow_sigmas:
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=self.image_seq_len, num_steps=S)
        timesteps, _n = retrieve_timesteps(sch, S, self.device, sigmas=sigmas, mu=mu)
        return timesteps, sch


def denoise(rec, latents, start, stop, hook=None):
    """Steps start+1 .. stop of rec's S-step schedule (0 <= start <= stop <= S), from `latents`
    (the input of step start+1: rec.latents0 when start = 0, a cached latent otherwise). Returns
    the latent after step `stop`.

    The body is Flux2KleinPipeline.__call__'s denoising loop line for line (timestep cast to the
    latents' dtype then / 1000, cond and uncond passes under their cache contexts, the CFG
    combination in the transformer's dtype, scheduler.step). The scheduler's step index is the
    only state carried between steps; set_begin_index(start) puts it where a full run would have
    it, so a run stopped at t and resumed from its latent is the full run.

    hook(step, latents_in, velocity, sigma, latents_out) fires after every step (step = 1-based
    index of the step just taken; latents_in its input; velocity the guided prediction).
    """
    import torch
    pipe = rec.pipe
    tf = pipe.transformer
    with torch.no_grad():
        timesteps, sch = rec.schedule()
        sch.set_begin_index(int(start))
        for i in range(int(start), int(stop)):
            t = timesteps[i]
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents.to(tf.dtype)
            with tf.cache_context("cond"):
                noise_pred = tf(hidden_states=latent_model_input, timestep=timestep / 1000, guidance=None,
                                encoder_hidden_states=rec.emb, txt_ids=rec.text_ids, img_ids=rec.latent_ids,
                                joint_attention_kwargs=None, return_dict=False)[0]
            noise_pred = noise_pred[:, : latents.size(1):]
            if rec.cfg:
                with tf.cache_context("uncond"):
                    neg_noise_pred = tf(hidden_states=latent_model_input, timestep=timestep / 1000, guidance=None,
                                        encoder_hidden_states=rec.neg, txt_ids=rec.neg_ids, img_ids=rec.latent_ids,
                                        joint_attention_kwargs=None, return_dict=False)[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1):]
                noise_pred = neg_noise_pred + rec.guidance * (noise_pred - neg_noise_pred)
            latents_in = latents
            latents = sch.step(noise_pred, t, latents, return_dict=False)[0]
            if hook is not None:
                hook(i + 1, latents_in, noise_pred, sch.sigmas[i], latents)
    return latents


def x0hat(latents_in, velocity, sigma):
    """x̂0 = x_t − σ_t v̂_t in fp32, cast back to the latents' dtype so it enters the decode exactly
    where the pipeline's final latents would."""
    x0 = latents_in.float() - sigma.float().to(latents_in.device) * velocity.float()
    return x0.to(latents_in.dtype)


def decode_pil(pipe, latents, latent_ids):
    """Flux2KleinPipeline.__call__'s tail for output_type="pil", verbatim."""
    import torch
    with torch.no_grad():
        lat = pipe._unpack_latents_with_ids(latents, latent_ids)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            lat.device, lat.dtype)
        lat = lat * bn_std + bn_mean
        lat = pipe._unpatchify_latents(lat)
        image = pipe.vae.decode(lat, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]


def dino_embed(dino, dino_transform, image, device):
    """The tool's DINOv2 readout (gpu_pool._process_discover): transform -> ViT-B/14-reg CLS ->
    L2-normalised; returns a (768,) float32 numpy vector."""
    import torch
    with torch.no_grad():
        e = dino(dino_transform(image).unsqueeze(0).to(device))
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy().flatten().astype(np.float32)


def x0hat_embedding(rec, dino, dino_transform, latents_in, velocity, sigma):
    """(PIL image, DINOv2 embedding) of the predicted clean image of one step."""
    img = decode_pil(rec.pipe, x0hat(latents_in, velocity, sigma), rec.latent_ids)
    return img, dino_embed(dino, dino_transform, img, rec.device)


def jpeg(image, quality=90):
    """gpu_pool._image_to_thumbnail: full-resolution JPEG, as every Cascade image is stored."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_basis(pipe, prompts, device=None):
    import torch
    with torch.no_grad():
        return [pipe.encode_prompt(prompt=p)[0] if device is None else pipe.encode_prompt(prompt=p, device=device)[0]
                for p in prompts]


# ---- the four worker bodies -----------------------------------------------------------------

def worker_readout(pipe, dino, dino_transform, device, base, weights, seed, height, width, steps, t, guidance):
    """Steps 1..t of the full schedule; DINOv2 of x̂0 at step t; the latent after step t."""
    rec = StagedRecipe(pipe, mix_discover(base, weights), seed, height, width, steps, guidance, device)
    got = {}
    sw = Stopwatch(device)

    def hook(step, lat_in, v, sigma, _lat_out):
        if step == t:
            got["den"] = sw.pause()
            t0 = time.perf_counter()
            img, e = x0hat_embedding(rec, dino, dino_transform, lat_in, v, sigma)
            _sync(device)
            got["dd"] = time.perf_counter() - t0
            got["img"], got["emb"] = img, e
            sw.resume()

    lat = denoise(rec, rec.latents0, 0, t, hook)
    b, shape, dt = latent_to_bytes(lat)
    return {"emb": got["emb"], "thumb": jpeg(got["img"]), "latent": b, "shape": shape, "dtype": dt,
            "timing": {"denoise_s": float(got["den"]), "decode_dino_s": float(got["dd"]), "steps": int(t)}}


def worker_resume(pipe, dino, dino_transform, device, base, weights, seed, height, width, steps, t, guidance,
                  latent=None, shape=None, dtype=None):
    """Steps t+1..S from a cached latent (latent None: the whole run from noise -- the same image,
    at full price); the final image decoded exactly as the pipeline does, and its DINOv2."""
    rec = StagedRecipe(pipe, mix_discover(base, weights), seed, height, width, steps, guidance, device)
    scratch = latent is None
    start = 0 if scratch else int(t)
    lat_in = rec.latents0 if scratch else bytes_to_latent(latent, shape, dtype, device)
    sw = Stopwatch(device)
    lat = denoise(rec, lat_in, start, rec.steps)
    den = sw.pause()
    t0 = time.perf_counter()
    img = decode_pil(pipe, lat, rec.latent_ids)
    e = dino_embed(dino, dino_transform, img, device)
    _sync(device)
    dd = time.perf_counter() - t0
    return {"emb": e, "thumb": jpeg(img),
            "timing": {"denoise_s": float(den), "decode_dino_s": float(dd), "steps": int(rec.steps - start),
                       "from_scratch": bool(scratch)}}


def worker_trace_point(pipe, dino, dino_transform, device, base, weights, seed, height, width, steps, guidance,
                       fourstep=None):
    """h27's per-point record: the full S-step run with DINOv2 of x̂0 after every step 1..S-1 and
    of the final image (t = S); per-step cumulative denoising time; per-readout decode+DINOv2
    time; and, when `fourstep(weights) -> (PIL, emb)` is given (the Cascade probe's own code
    path), DINOv2 of the complete 4-step-schedule image and its wall-clock."""
    rec = StagedRecipe(pipe, mix_discover(base, weights), seed, height, width, steps, guidance, device)
    S = rec.steps
    embs = [None] * S
    cum = [0.0] * S
    dd = [0.0] * S
    sw = Stopwatch(device)

    def hook(step, lat_in, v, sigma, _lat_out):
        cum[step - 1] = sw.pause()
        if step < S:
            t0 = time.perf_counter()
            _img, e = x0hat_embedding(rec, dino, dino_transform, lat_in, v, sigma)
            _sync(device)
            dd[step - 1] = time.perf_counter() - t0
            embs[step - 1] = e
        sw.resume()

    lat = denoise(rec, rec.latents0, 0, S, hook)
    sw.pause()
    t0 = time.perf_counter()
    img = decode_pil(pipe, lat, rec.latent_ids)
    embs[S - 1] = dino_embed(dino, dino_transform, img, device)
    _sync(device)
    dd[S - 1] = time.perf_counter() - t0
    out = {"xhat": np.stack(embs).astype(np.float32), "denoise_cum_s": cum, "decode_dino_s": dd}
    if fourstep is not None:
        _sync(device)
        t0 = time.perf_counter()
        _img4, e4 = fourstep(weights)
        _sync(device)
        out["fourstep"] = np.asarray(e4, dtype=np.float32)
        out["fourstep_s"] = time.perf_counter() - t0
    return out


def worker_onepass(pipe, device, base, weights_list, seed, height, width, batch_size=8):
    """The Fast-Scan readout of gpu_pool._ScanContext, generalised from (alpha, beta[, gamma]) to
    any k-weight vector and nothing else: noise from prepare_latents with the seeded device
    generator, ONE transformer call at the first timestep of a 1-step schedule (mu for
    num_steps=1), no CFG, x̂0 = noise − velocity, flattened, L2-normalised, batches of 8 in the
    given order, embeddings mixed by gpu_pool._mix_embeddings (LERP). Returns (N, D) float32.
    The text ids are the first prompt's, as _ScanContext takes prompt_a's."""
    import torch
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
    from backend.services.gpu_pool import _mix_embeddings
    with torch.no_grad():
        embs, text_ids = [], None
        for p in base:
            e, ti = pipe.encode_prompt(prompt=p)
            embs.append(e)
            text_ids = ti if text_ids is None else text_ids
        gen = torch.Generator(device=device).manual_seed(int(seed))
        in_ch = pipe.transformer.config.in_channels
        noise, latent_ids = pipe.prepare_latents(
            batch_size=1, num_latents_channels=in_ch // 4, height=height, width=width,
            dtype=embs[0].dtype, device=device, generator=gen)
        img_tokens = noise.shape[1]
        mu = compute_empirical_mu(image_seq_len=img_tokens, num_steps=1)
        pipe.scheduler.set_timesteps(num_inference_steps=1, device=device, mu=mu)
        t_val = pipe.scheduler.timesteps[0]
        tf = pipe.transformer
        noise_t = noise.to(tf.dtype)
        batch_t = (t_val / 1000).to(tf.dtype)
        out = []
        for b0 in range(0, len(weights_list), batch_size):
            batch = weights_list[b0:b0 + batch_size]
            bs = len(batch)
            batch_embeds = torch.cat([_mix_embeddings(embs, [float(x) for x in w], False) for w in batch],
                                     dim=0).to(noise_t.dtype)
            batch_noise = noise_t.expand(bs, -1, -1)
            velocity = tf(hidden_states=batch_noise, timestep=batch_t.expand(bs), guidance=None,
                          encoder_hidden_states=batch_embeds, txt_ids=text_ids.expand(bs, -1, -1),
                          img_ids=latent_ids.expand(bs, -1, -1), return_dict=False)[0]
            velocity = velocity[:, :img_tokens, :]
            denoised = batch_noise - velocity
            for idx in range(bs):
                flat = denoised[idx].flatten().float()
                norm = flat.norm()
                if norm > 1e-8:
                    flat = flat / norm
                out.append(flat.cpu().numpy())
    return np.stack(out).astype(np.float32)


def worker_parity(pipe, dino, dino_transform, device, base, weights, seed, height, width, steps, t, guidance,
                  pipe_check=True):
    """V1 for one point: steps 1..t -> latent -> bytes -> latent -> steps t+1..S, against the
    direct S-step run of the same loop and (pipe_check) against pipe() itself, the Cascade's
    image path (same args as gpu_pool._process_discover, final packed latents captured by a
    step-end callback)."""
    import torch
    emb = mix_discover(base, weights)
    rec = StagedRecipe(pipe, emb, seed, height, width, steps, guidance, device)
    S = rec.steps
    sw = Stopwatch(device)
    direct = denoise(rec, rec.latents0, 0, S)
    t_direct = sw.pause()
    sw = Stopwatch(device)
    lat_t = denoise(rec, rec.latents0, 0, t)
    t_read = sw.pause()
    b, shape, dt = latent_to_bytes(lat_t)
    sw = Stopwatch(device)
    resumed = denoise(rec, bytes_to_latent(b, shape, dt, device), t, S)
    t_res = sw.pause()
    img_d = decode_pil(pipe, direct, rec.latent_ids)
    img_r = decode_pil(pipe, resumed, rec.latent_ids)
    e_d = dino_embed(dino, dino_transform, img_d, device)
    e_r = dino_embed(dino, dino_transform, img_r, device)
    out = {"cos_resumed_vs_direct": float(np.dot(e_d, e_r)),
           "max_abs_latent_diff": float((resumed.float() - direct.float()).abs().max()),
           "latent_bit_identical": bool(torch.equal(resumed, direct)),
           "pixels_identical": bool(np.array_equal(np.asarray(img_d), np.asarray(img_r))),
           "latent_bytes": len(b), "latent_shape": list(shape), "latent_dtype": dt,
           "timing": {"direct_s": t_direct, "readout_s": t_read, "resume_s": t_res}}
    if pipe_check:
        cap = {}

        def cb(_p, i, _t, kw):
            cap["lat"] = kw["latents"].clone()
            return kw
        gen = torch.Generator(device=device).manual_seed(int(seed))
        with torch.no_grad():
            img_p = pipe(prompt_embeds=emb, height=height, width=width, num_inference_steps=S,
                         negative_prompt_embeds=getattr(pipe, "_cached_neg_embeds", None),
                         guidance_scale=guidance, generator=gen,
                         callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
        e_p = dino_embed(dino, dino_transform, img_p, device)
        out.update({"cos_resumed_vs_pipe": float(np.dot(e_p, e_r)),
                    "max_abs_latent_diff_pipe": float((resumed.float() - cap["lat"].float()).abs().max()),
                    "pixels_identical_pipe": bool(np.array_equal(np.asarray(img_p), np.asarray(img_r)))})
    return out


# =============================================================================================
# MAIN-process side
# =============================================================================================

def latent_key(prompts, weights, seed, steps, t, guidance, height, width):
    """The cache key of one probe's latent: the full recipe + seed + S + t + guidance + size."""
    return (tuple(str(p) for p in prompts), tuple(round(float(w), 9) for w in weights), int(seed), int(steps),
            int(t), round(float(guidance), 6), int(height), int(width))


class LatentCache:
    """LRU of cached latents (raw bytes + shape + dtype) under a byte cap. Thread-safe: Cascade
    runs, desks and the research endpoints share one instance per process."""

    def __init__(self, cap_bytes=CACHE_CAP_BYTES):
        self.cap = int(cap_bytes)
        self._d = OrderedDict()
        self.nbytes = 0
        self.hits = self.misses = self.evictions = 0
        self._lock = threading.Lock()

    def put(self, key, data, shape, dtype):
        n = len(data)
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.nbytes -= len(old[0])
            if n > self.cap:
                return False                      # never cache what cannot fit at all
            self._d[key] = (data, tuple(shape), str(dtype))
            self.nbytes += n
            while self.nbytes > self.cap and self._d:
                _k, (b, _s, _t) = self._d.popitem(last=False)
                self.nbytes -= len(b)
                self.evictions += 1
            return True

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return v

    def __contains__(self, key):
        with self._lock:
            return key in self._d

    def __len__(self):
        with self._lock:
            return len(self._d)

    def stats(self):
        with self._lock:
            return {"entries": len(self._d), "bytes": int(self.nbytes), "cap_bytes": int(self.cap),
                    "hits": int(self.hits), "misses": int(self.misses), "evictions": int(self.evictions)}


_CACHE = LatentCache()


def get_cache(app=None):
    """app.state.staged_cache when set (tests use small caps), else the process-wide instance."""
    st = getattr(app, "state", None) if app is not None else None
    c = getattr(st, "staged_cache", None) if st is not None else None
    return c if c is not None else _CACHE


# ---- pure rules -------------------------------------------------------------------------------

def _cosd(a, b):
    return 1.0 - float(np.dot(a, b))


def _norm(e):
    e = np.asarray(e, dtype=np.float64)
    n = np.linalg.norm(e)
    return e / n if n > 0 else e


def segment_divs(embs):
    """1 − cos between consecutive readouts (None where either end is missing)."""
    return [None if embs[i] is None or embs[i + 1] is None else _cosd(embs[i], embs[i + 1])
            for i in range(len(embs) - 1)]


def flag_segments(embs, theta, skip_first=False):
    """A segment is FLAGGED when its two x̂0 readouts are at least theta apart. A segment with a
    missing readout is no evidence either way and is not flagged (the steps mode treats a missing
    probe the same). skip_first drops segment 0 (a branching child ray starts on its parent's
    crossing; the Cascade's detection skips that bracket too)."""
    out = []
    for i, d in enumerate(segment_divs(embs)):
        out.append(bool(d is not None and d >= float(theta) and not (skip_first and i == 0)))
    return out


def resume_selection(seqs, flags, done=()):
    """The probes to resume: both ends of every flagged segment, each once, in first-seen order.

    seqs  -- per sequence (chord / line), its probes' ids in order (None = never read out)
    flags -- per sequence, one bool per consecutive pair (flag_segments)
    done  -- ids that already carry a full-fidelity label (skipped)
    """
    seen = set(done)
    out = []
    for seq, fl in zip(seqs, flags):
        for j, f in enumerate(fl):
            if not f:
                continue
            for g in (seq[j], seq[j + 1]):
                if g is not None and g not in seen:
                    seen.add(g)
                    out.append(g)
    return out


class Ledger:
    """Cost of a run's staged probes, in image-eq (1 = one full S-step image + its decode and
    DINOv2), from the workers' measured timings.

    nominal  -- readout t/S, resume (S−t)/S, a from-scratch resume 1.
    measured -- each operation's own (denoise + decode/DINOv2) seconds over the run's current
                estimate of one full image, S x (seconds per step) + (seconds per decode+DINOv2).
                So a readout costs t/S of the steps plus one decode+DINOv2 overhead, and readout +
                resume costs one image plus one extra decode.
    """

    def __init__(self, steps, t):
        self.S, self.t = int(steps), int(t)
        self.n_readout = self.n_resume = self.n_scratch = 0
        self.den_s = self.dd_s = 0.0
        self.n_steps = self.n_dd = 0
        self.image_eq = 0.0
        self.image_eq_nominal = 0.0
        self.segments = self.flagged = 0
        self.rebracket_skipped = 0
        self.cache_misses = 0

    def unit_s(self):
        if self.n_steps == 0 or self.n_dd == 0:
            return None
        return self.S * (self.den_s / self.n_steps) + self.dd_s / self.n_dd

    def add(self, kind, timing):
        """Charge one operation; returns its measured image-eq (nominal until timings exist)."""
        steps = int(timing.get("steps", 0))
        den = float(timing.get("denoise_s", 0.0))
        dd = float(timing.get("decode_dino_s", 0.0))
        scratch = bool(timing.get("from_scratch", False))
        if kind == "readout":
            self.n_readout += 1
            nom = self.t / self.S
        else:
            self.n_resume += 1
            if scratch:
                self.n_scratch += 1
            nom = 1.0 if scratch else (self.S - self.t) / self.S
        self.image_eq_nominal += nom
        if steps > 0 and den > 0:
            self.den_s += den
            self.n_steps += steps
        if dd > 0:
            self.dd_s += dd
            self.n_dd += 1
        u = self.unit_s()
        cost = (den + dd) / u if (u and (den > 0 or dd > 0)) else nom
        self.image_eq += cost
        return cost

    def report(self):
        u = self.unit_s()
        return {"t": self.t, "steps": self.S, "readouts": self.n_readout, "resumed": self.n_resume - self.n_scratch,
                "from_scratch": self.n_scratch,
                "resumed_share": (self.n_resume / self.n_readout) if self.n_readout else None,
                "segments": self.segments, "flagged": self.flagged,
                "flagged_share": (self.flagged / self.segments) if self.segments else None,
                "image_eq": round(self.image_eq, 4), "image_eq_nominal": round(self.image_eq_nominal, 4),
                "image_eq_per_readout": round(self.image_eq / self.n_readout, 4) if self.n_readout else None,
                "decode_share": round((self.dd_s / self.n_dd) / u, 4) if u else None,
                "unit_s": round(u, 4) if u else None,
                "rebracket_skipped": self.rebracket_skipped, "cache_misses": self.cache_misses}


def ledger(run):
    """The run's Ledger, created on first use (CascadeRun / DeskRun carry `staged_ledger`)."""
    lg = getattr(run, "staged_ledger", None)
    if lg is None:
        lg = Ledger(run.steps, getattr(run, "staged_t", T_DEFAULT))
        run.staged_ledger = lg
    return lg


# ---- dispatch (cascade.evaluate's shard / inbox discipline) ----------------------------------

def _dispatch(app, pool, ctl, items, make_task, on_result, label, run_id="x", n_gpus=None):
    """Submit `items` [(index, payload)] in rounds of n_gpus x CHUNK, sharded round-robin over
    the GPUs, and feed every StagedResult that comes back to on_result(result). Mirrors
    cascade.evaluate: inbox registered before submit, cooperative stop through ctl.status, a
    pool-wide cancel marks ctl cancelled, worker errors and stalls end the batch with a note.
    n_gpus overrides the pool's worker count (1 = every item on one worker, in order).
    Returns the set of indices that arrived."""
    if not items:
        return set()
    ep = pool.cancel_epoch.value
    n_gpus = max(1, min(n_gpus or getattr(pool, "n_gpus", 1), len(items)))
    step = n_gpus * CHUNK
    got_idx = set()
    seq = _next_seq()
    for off in range(0, len(items), step):
        if ctl.status != "running":
            break
        block = items[off:off + step]
        shards = []
        for g in range(n_gpus):
            shard = block[g::n_gpus]
            if not shard:
                continue
            tid = f"staged:{run_id}:{label}.{seq}:{off}:{g}"
            app.state.hike_inbox.setdefault(tid, [])
            shards.append((tid, len(shard)))
            pool.submit(make_task(tid, shard))
        want = sum(n for _, n in shards)
        got, last = 0, time.time()
        while got < want and ctl.status == "running":
            moved = False
            for tid, _n in shards:
                ib = app.state.hike_inbox.get(tid) or []
                while ib:
                    r = ib.pop(0)
                    if getattr(r, "error", ""):
                        ctl.notes.append(f"staged {label}: point {r.index} failed: {r.error}")
                    else:
                        try:
                            on_result(r)
                            got_idx.add(r.index)
                        except Exception as exc:          # noqa: BLE001
                            ctl.notes.append(f"staged {label}: result {r.index} dropped ({exc})")
                    got += 1
                    moved = True
            if moved:
                last = time.time()
                continue
            if pool.cancel_epoch.value != ep:
                ctl.notes.append("cancelled by a pool-wide cancel")
                ctl.status = "cancelled"
                break
            he = getattr(app.state, "hike_task_errors", {})
            err = next((he.pop(tid) for tid, _n in shards if tid in he), None)
            if err is not None:
                ctl.notes.append(f"worker error: {err}")
                break
            if time.time() - last > STALL_TIMEOUT:
                ctl.notes.append(f"staged {label} stalled at {got}/{want}; moving on")
                break
            time.sleep(0.1)
        for tid, _n in shards:
            app.state.hike_inbox.pop(tid, None)
    return got_idx


def readout(app, run, pool, weights, label, ctl=None, on_arrival=None):
    """Staged readout of every weight vector: steps 1..t of the run's full schedule, DINOv2 of x̂0
    at t, the latent after step t into the LRU. Returns one global index per input (None where
    it never arrived).

    Run-side effects per arrival: run.xhat[gi] (unit x̂0 embedding), run.staged_w[gi] (the
    recipe), run.staged_lat[gi] (its cache key), run.thumbs[gi] = the x̂0 PREVIEW (listed in
    run.staged_preview until a resume overwrites it with the finished image), the map cloud
    (probe_geo / probe_div) and phase_done. run.embeddings is NOT touched: it holds labels, and
    an x̂0 readout is not one. Nothing is charged to run.generated (a readout is not an image).
    """
    from backend.services.gpu_pool import StagedReadoutTask
    if not weights:
        return []
    ctl = ctl if ctl is not None else run
    with run._lock:
        idxs = list(range(run.next_idx, run.next_idx + len(weights)))
        run.next_idx += len(weights)
    local = {gi: li for li, gi in enumerate(idxs)}
    pts = {gi: [float(x) for x in w] for gi, w in zip(idxs, weights)}
    cache = get_cache(app)
    lg = ledger(run)
    t = int(run.staged_t)

    def make(tid, shard):
        return StagedReadoutTask(job_id=tid, basis=list(run.prompts), points=shard, seed=run.seed,
                                 height=run.height, width=run.width, steps=run.steps, t=t,
                                 guidance_scale=run.guidance_scale)

    def on_result(r):
        gi = r.index
        if gi not in pts:
            return
        d = r.data
        key = latent_key(run.prompts, pts[gi], run.seed, run.steps, t, run.guidance_scale, run.height, run.width)
        cache.put(key, d["latent"], d["shape"], d["dtype"])
        run.staged_lat[gi] = key
        run.staged_w[gi] = pts[gi]
        run.xhat[gi] = _norm(d["emb"])
        if run.thumbs is not None and d.get("thumb"):
            run.thumbs[gi] = d["thumb"]
            run.staged_preview.add(gi)
        lg.add("readout", d.get("timing", {}))
        if len(run.probe_geo) < 6000 and gi not in run._geo_pos:
            run._geo_pos[gi] = len(run.probe_geo)
            run.probe_geo.append(pts[gi])
            run.probe_div.append(None)
        run.phase_done += 1
        if on_arrival is not None:
            try:
                on_arrival(local[gi], gi)
            except Exception:                 # noqa: BLE001  colouring must never sink a run
                pass

    arrived = _dispatch(app, pool, ctl, [(gi, pts[gi]) for gi in idxs], make, on_result, label,
                        getattr(run, "run_id", "x"))
    return [gi if gi in arrived else None for gi in idxs]


def resume(app, run, pool, gis, label, ctl=None, on_arrival=None):
    """Finish the given read-out probes: steps t+1..S from their cached latents, the final image
    and its DINOv2 -- the exact full-fidelity label. A latent the LRU has evicted is re-run from
    noise (the same image at full price, counted as from_scratch). Returns the set of indices
    that now carry a label.

    Run-side effects per arrival: run.embeddings[gi] = the full-fidelity label, run.thumbs[gi] =
    the finished image (replacing the x̂0 preview), run.generated / recent_thumbs / phase_done.
    """
    from backend.services.gpu_pool import StagedResumeTask
    gis = [g for g in gis if g is not None and g in run.staged_w]
    if not gis:
        return set()
    ctl = ctl if ctl is not None else run
    cache = get_cache(app)
    lg = ledger(run)
    t = int(run.staged_t)
    items = []
    for gi in gis:
        hit = cache.get(run.staged_lat.get(gi))
        if hit is None:
            lg.cache_misses += 1
            items.append((gi, (run.staged_w[gi], None, None, None)))
        else:
            items.append((gi, (run.staged_w[gi],) + tuple(hit)))
    local = {gi: li for li, gi in enumerate(gis)}

    def make(tid, shard):
        return StagedResumeTask(job_id=tid, basis=list(run.prompts), points=shard, seed=run.seed,
                                height=run.height, width=run.width, steps=run.steps, t=t,
                                guidance_scale=run.guidance_scale)

    def on_result(r):
        gi = r.index
        d = r.data
        run.embeddings[gi] = _norm(d["emb"])
        if run.thumbs is not None and d.get("thumb"):
            run.thumbs[gi] = d["thumb"]
        run.staged_preview.discard(gi)
        run.recent_thumbs.append(gi)
        del run.recent_thumbs[:-12]
        lg.add("resume", d.get("timing", {}))
        run.generated += 1
        run.phase_done += 1
        if on_arrival is not None:
            try:
                on_arrival(local[gi], gi)
            except Exception:                 # noqa: BLE001
                pass

    return _dispatch(app, pool, ctl, items, make, on_result, label, getattr(run, "run_id", "x"))


def run_sequences(app, run, pool, gis_seqs, label, ctl=None, skip_first=None, always=(), on_resume=None):
    """The staged decision on already-read-out sequences: flag, select, resume.

    gis_seqs   -- per sequence, its probes' global indices in order (None = no readout)
    skip_first -- per sequence, drop segment 0 from flagging (child rays); None = never
    always     -- extra indices to resume whatever the flags say (the desk's current mix)
    Returns (flags per sequence, the resumed index list). Updates the ledger's segment counts.
    """
    lg = ledger(run)
    theta = float(run.staged_theta)
    flags = []
    for si, seq in enumerate(gis_seqs):
        embs = [run.xhat.get(g) if g is not None else None for g in seq]
        fl = flag_segments(embs, theta, bool(skip_first[si]) if skip_first is not None else False)
        flags.append(fl)
        lg.segments += sum(1 for i, d in enumerate(segment_divs(embs))
                           if d is not None and not (skip_first is not None and skip_first[si] and i == 0))
        lg.flagged += sum(fl)
    done = {g for g in run.embeddings}
    sel = resume_selection(gis_seqs, flags, done)
    for g in always:
        if g is not None and g not in done and g not in sel:
            sel.append(g)
    run.phase_total += len(sel)
    if sel:
        resume(app, run, pool, sel, label, ctl, on_resume)   # module global: a test's stub replaces it
    return flags, sel
