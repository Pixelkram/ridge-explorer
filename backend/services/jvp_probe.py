"""Grid-free local probes via exact forward-mode JVPs through the generator.

Two measurements that survived the h07/h08 gates (search_problem/outputs/h07_chain_SUMMARY.md,
h08_vector_uses/RESULTS.md), computed at ONE point of the prompt simplex without a grid:

  ridge_normal(...)   the direction in (alpha, beta[, gamma]) along which the image changes fastest —
                      the ridge normal. Whitened by the Gram matrix of the prompt embeddings so the text
                      encoder's own anisotropy is removed. Seed-invariant (cross-seed |cos| 0.93 raw /
                      0.91 whitened, out of sample), aligned with the grid's own normal. Also returns the
                      whitened spectrum: rank-1 share (~0.99 on a front) and participation ratio (~1 front,
                      ~2 corner). sigma1 is comparable ONLY between points of the same job — a global
                      "ridge / not ridge" threshold was refuted (h08-5).
  token_sensitivity() one JVP per token position of the prompt embedding: which word the generation is
                      load-bearing on at this point. Rankings are seed-invariant (Spearman 0.66) and follow
                      the word, not its slot (h08-6). Nothing here says HOW the image changes — that
                      direction is seed noise (h07b-d).

Runs inside a gpu_pool worker on its resident pipe + DINOv2. The pool's pipe is unpatched bf16; forward AD
needs manual-norm forwards and an fp32 VAE, so both are applied around the call and restored afterwards
(the tool's own generation is left bit-identical). Mixing (LERP / NLERP) and steps / guidance / size follow
the job, so the probe differentiates the same map the grid sampled.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/student/ai/search_problem")  # fisher.metric.jvp_exact + norm patches
from fisher.flux_pipeline import _patch_norms_bf16_safe, unpatch_norms  # noqa: E402
from fisher.metric import jvp_exact, math_attention  # noqa: E402
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps  # noqa: E402

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@contextmanager
def _forward_ad_ready(pipe):
    """Patch norms for forward AD, cast the VAE to fp32 and park the text encoder on the CPU (the pool keeps
    Qwen3-4B resident; forward AD needs ~3 GB of headroom the worker does not otherwise have). Restore all
    three on exit so the worker's ordinary generation is unchanged."""
    device = pipe.transformer.device
    _patch_norms_bf16_safe(pipe.transformer)
    vae_dtype = next(pipe.vae.parameters()).dtype
    pipe.vae.float()
    te = getattr(pipe, "text_encoder", None)
    if te is not None:
        te.to("cpu")
    # Freeze weights for the duration: with requires_grad left True (the pool never trains, but from_pretrained
    # leaves it on) reverse-mode autograd records the whole 20-pass graph alongside the forward-mode tangent and
    # the probe OOMs at ~23 GB instead of ~11 GB. Flags are restored on exit.
    frozen = []
    for m in (pipe.transformer, pipe.vae):
        for prm in m.parameters():
            if prm.requires_grad:
                frozen.append(prm); prm.requires_grad_(False)
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        for prm in frozen:
            prm.requires_grad_(True)
        unpatch_norms(pipe.transformer)
        pipe.vae.to(vae_dtype)
        if te is not None:
            te.to(device)
        torch.cuda.empty_cache()


def _unpack(x, x_ids):
    pos = x_ids[0]
    h_ids, w_ids = pos[:, 1].long(), pos[:, 2].long()
    h, w = int(h_ids.max()) + 1, int(w_ids.max()) + 1
    order = torch.argsort(h_ids * w + w_ids)
    return x[0][order].view(h, w, -1).permute(2, 0, 1).unsqueeze(0)


def _unpatchify(lat):
    b, c, h, w = lat.shape
    return lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


def _make_tail(pipe, dino, latent_ids):
    vae = pipe.vae
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).float()
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).float()
    mean, std = _MEAN.to(vae.device), _STD.to(vae.device)

    def tail(z0):
        lat = _unpack(z0.float(), latent_ids) * bn_std + bn_mean
        img = vae.decode(_unpatchify(lat), return_dict=False)[0]
        img = (img / 2 + 0.5).clamp(0, 1)
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        feat = dino((img - mean) / std)
        return feat / feat.norm(dim=-1, keepdim=True)
    return tail


def _mix_differentiable(embs, weights, use_slerp):
    """gpu_pool._mix_embeddings with tensor weights (theta) so forward AD flows through the mixing."""
    result = sum(w * e for w, e in zip(weights, embs))
    if not use_slerp:
        return result
    target_norm = sum(w * e.norm() for w, e in zip(weights, embs))
    result_norm = result.norm()
    return torch.where((result_norm > 1e-8) & (target_norm > 1e-8), result * (target_norm / result_norm), result)


def _neg_embeds(pipe, device):
    neg = getattr(pipe, "_cached_neg_embeds", None)
    if neg is None:
        with torch.no_grad():
            neg, _ = pipe.encode_prompt(prompt="", device=device)
    return neg, pipe._prepare_text_ids(neg).to(device)


def _make_map(pipe, dino, embs, text_ids, z_T, latent_ids, *, steps, guidance, use_slerp, neg=None, neg_ids=None):
    """theta (k-1,) fp32 -> unit DINOv2 CLS. embs[0] is the remainder prompt; theta = weights of embs[1:]."""
    tf, sch, device = pipe.transformer, pipe.scheduler, pipe.transformer.device
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    if getattr(sch.config, "use_flow_sigmas", False):
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=z_T.shape[1], num_steps=steps)
    timesteps, _ = retrieve_timesteps(sch, steps, device, sigmas=sigmas, mu=mu)
    timesteps = timesteps.clone()
    tail = _make_tail(pipe, dino, latent_ids)
    E = [e.float() for e in embs]
    cfg = guidance > 1.0

    def g(theta):
        w = [1 - theta.sum()] + [theta[i] for i in range(theta.numel())]
        emb = _mix_differentiable(E, w, use_slerp).to(tf.dtype)
        lat = z_T
        sch._step_index = None
        sch.set_begin_index(0)
        for t in timesteps:
            ts = t.expand(1).to(lat.dtype)
            pos = tf(hidden_states=lat.to(tf.dtype), timestep=ts / 1000, guidance=None, encoder_hidden_states=emb,
                     txt_ids=text_ids, img_ids=latent_ids, return_dict=False)[0][:, : lat.size(1)]
            if cfg:
                ng = tf(hidden_states=lat.to(tf.dtype), timestep=ts / 1000, guidance=None, encoder_hidden_states=neg,
                        txt_ids=neg_ids, img_ids=latent_ids, return_dict=False)[0][:, : lat.size(1)]
                pos = ng + guidance * (pos - ng)
            lat = sch.step(pos, t, lat, return_dict=False)[0]
        return tail(lat)[0]
    return g


def _noise(pipe, seed, height, width):
    device = pipe.transformer.device
    gen = torch.Generator(device=device).manual_seed(seed)
    return pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, height, width, torch.bfloat16, device, gen, None)


def _gram(embs):
    E = [e[0].float().reshape(-1).double().cpu().numpy() for e in embs]
    D = [E[i] - E[0] for i in range(1, len(E))]
    return np.array([[a @ b for b in D] for a in D])


def _freeze(dino):
    for prm in dino.parameters():
        prm.requires_grad_(False)


def ridge_normal(pipe, dino, embs, text_ids, weights, *, seed, steps, guidance, height, width, use_slerp):
    """Exact local Jacobian at barycentric `weights` (k entries, sum 1; weights[0] = remainder prompt).

    Returns the whitened normal as a k-vector of barycentric deltas (sum 0, unit in whitened metric),
    the same in raw (alpha, beta, ...) coordinates, the whitened spectrum, rank-1 share, participation
    ratio, sigma1, and wall time. embs: list of k prompt embeddings (1, 512, D) bf16.
    """
    device = pipe.transformer.device
    _freeze(dino)
    k = len(embs)
    theta = torch.tensor(weights[1:], dtype=torch.float32, device=device)
    z_T, latent_ids = _noise(pipe, seed, height, width)
    neg, neg_ids = _neg_embeds(pipe, device) if guidance > 1.0 else (None, None)
    t0 = time.time()
    with _forward_ad_ready(pipe):
        g = _make_map(pipe, dino, embs, text_ids, z_T, latent_ids, steps=steps, guidance=guidance, use_slerp=use_slerp, neg=neg, neg_ids=neg_ids)
        cols = []
        for i in range(k - 1):
            v = torch.zeros(k - 1, device=device)
            v[i] = 1.0
            cols.append(jvp_exact(g, theta, v).detach().float().cpu().numpy())
    J = np.stack(cols, 1).astype(np.float64)                       # (768, k-1), raw coordinates
    G = _gram(embs)
    ev, Q = np.linalg.eigh(G)
    Whi = np.linalg.inv(Q @ np.diag(np.sqrt(np.clip(ev, 1e-12, None))) @ Q.T)
    U, s, Vt = np.linalg.svd(J @ Whi, full_matrices=False)          # whitened spectrum
    n_theta = Whi @ Vt[0]                                           # steepest direction back in theta space
    n_theta /= np.linalg.norm(n_theta)
    s2 = s ** 2
    n_bary = np.concatenate([[-n_theta.sum()], n_theta])            # barycentric deltas, sum 0
    sraw = np.linalg.svd(J, compute_uv=False)
    return {
        "normal_bary": n_bary.tolist(), "normal_theta": n_theta.tolist(),
        "sigma_whitened": s.tolist(), "sigma1_raw": float(sraw[0]),
        "rank1_share": float(s2[0] / s2.sum()), "participation_ratio": float(s2.sum() ** 2 / (s2 ** 2).sum()),
        "k": k, "steps": steps, "guidance": guidance, "seed": seed, "use_slerp": use_slerp, "wall_s": time.time() - t0,
    }


def token_positions(pipe, prompt, max_template=8, n_pad=3):
    """Content / template / pad positions in the pipeline's chat-templated, padded token sequence."""
    tok = pipe.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    ids, mask = enc["input_ids"][0].tolist(), enc["attention_mask"][0].tolist()
    raw = tok(prompt, add_special_tokens=False)["input_ids"]
    start = next(i for i in range(len(ids) - len(raw) + 1) if ids[i:i + len(raw)] == raw)
    content = list(range(start, start + len(raw)))
    template = [p for p in range(512) if mask[p] == 1 and p not in content][:max_template]
    pad = [p for p in range(512) if mask[p] == 0][:n_pad]
    return content, template, pad, {p: tok.decode([ids[p]]) for p in content + template + pad}


def token_sensitivity(pipe, dino, prompt, *, seed, steps, guidance, height, width, include_template=True):
    """One exact JVP per token position (removal direction -E[row]) of a single prompt. Returns rows sorted by
    position with sigma = ||J v|| / ||v||, plus the ranking of content tokens."""
    device = pipe.transformer.device
    _freeze(dino)
    with torch.no_grad():
        E, text_ids = pipe.encode_prompt(prompt=prompt, device=device)
    content, template, pad, text = token_positions(pipe, prompt)
    z_T, latent_ids = _noise(pipe, seed, height, width)
    neg, neg_ids = _neg_embeds(pipe, device) if guidance > 1.0 else (None, None)
    t0 = time.time()
    rows = []
    with _forward_ad_ready(pipe):
        g_of_E = _make_map_E(pipe, dino, text_ids, z_T, latent_ids, steps=steps, guidance=guidance, neg=neg, neg_ids=neg_ids)
        for cls, plist in (("content", content), ("template", template if include_template else []), ("pad", pad)):
            for q in plist:
                v = torch.zeros_like(E)
                v[0, q, :] = -E[0, q, :]
                vn = float(v.float().norm())
                Jv = jvp_exact(g_of_E, E, v).detach().float().cpu().numpy()
                rows.append({"pos": q, "cls": cls, "text": text[q], "sigma": float(np.linalg.norm(Jv) / vn)})
    content_rows = sorted((r for r in rows if r["cls"] == "content"), key=lambda r: -r["sigma"])
    return {"prompt": prompt, "rows": rows, "ranking": [r["text"] for r in content_rows], "seed": seed, "steps": steps,
            "guidance": guidance, "n_jvp": len(rows), "wall_s": time.time() - t0}


def _make_map_E(pipe, dino, text_ids, z_T, latent_ids, *, steps, guidance, neg=None, neg_ids=None):
    """E (1, 512, D) -> unit DINOv2 CLS; the map for token tangents."""
    tf, sch, device = pipe.transformer, pipe.scheduler, pipe.transformer.device
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    if getattr(sch.config, "use_flow_sigmas", False):
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=z_T.shape[1], num_steps=steps)
    timesteps, _ = retrieve_timesteps(sch, steps, device, sigmas=sigmas, mu=mu)
    timesteps = timesteps.clone()
    tail = _make_tail(pipe, dino, latent_ids)
    cfg = guidance > 1.0

    def g(E):
        lat = z_T
        sch._step_index = None
        sch.set_begin_index(0)
        for t in timesteps:
            ts = t.expand(1).to(lat.dtype)
            pos = tf(hidden_states=lat.to(tf.dtype), timestep=ts / 1000, guidance=None, encoder_hidden_states=E,
                     txt_ids=text_ids, img_ids=latent_ids, return_dict=False)[0][:, : lat.size(1)]
            if cfg:
                ng = tf(hidden_states=lat.to(tf.dtype), timestep=ts / 1000, guidance=None, encoder_hidden_states=neg,
                        txt_ids=neg_ids, img_ids=latent_ids, return_dict=False)[0][:, : lat.size(1)]
                pos = ng + guidance * (pos - ng)
            lat = sch.step(pos, t, lat, return_dict=False)[0]
        return tail(lat)[0]
    return g


def evaluate_plain(pipe, dino, embs, text_ids, weights, *, seed, steps, guidance, height, width, use_slerp):
    """The same map evaluated without AD (for tests): unit DINOv2 CLS at `weights`."""
    device = pipe.transformer.device
    theta = torch.tensor(weights[1:], dtype=torch.float32, device=device)
    z_T, latent_ids = _noise(pipe, seed, height, width)
    neg, neg_ids = _neg_embeds(pipe, device) if guidance > 1.0 else (None, None)
    with _forward_ad_ready(pipe), torch.no_grad(), math_attention():
        g = _make_map(pipe, dino, embs, text_ids, z_T, latent_ids, steps=steps, guidance=guidance, use_slerp=use_slerp, neg=neg, neg_ids=neg_ids)
        return g(theta).detach().float().cpu().numpy()
